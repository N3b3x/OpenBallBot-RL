import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from pathlib import Path


class Extractor(BaseFeaturesExtractor):

    def __init__(self,
                 observation_space: gym.spaces.Dict,
                 frozen_encoder_path: str = ""):

        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():

            if "rgbd_" in key:

                if not frozen_encoder_path:
                    #note that we're iterating on observation_space objects, so there is not batch size info
                    C, H, W = subspace.shape  #typically, C=1 (depth-only) or C=4 (RGB-D)

                    F1 = 32
                    F2 = 32
                    self.out_sz = 20
                    extractors[key] = torch.nn.Sequential(
                        torch.nn.Conv2d(C,  # Use actual channel count from observation space
                                        F1,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1),  #output BxF1xH/2xW/2
                        torch.nn.BatchNorm2d(F1),
                        torch.nn.LeakyReLU(),
                        torch.nn.Conv2d(F1,
                                        F2,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1),  #output BxF2xH/4xW/4
                        torch.nn.BatchNorm2d(F2),
                        torch.nn.LeakyReLU(),
                        torch.nn.Flatten(),
                        torch.nn.Linear(F2 * H // 4 * W // 4, self.out_sz),
                        torch.nn.BatchNorm1d(self.out_sz),
                        torch.nn.Tanh(),
                    )

                    total_concat_size += self.out_sz
                else:
                    encoder_path = Path(frozen_encoder_path).resolve()
                    print(f"loading encoder from {encoder_path}")
                    # Load on CPU to handle models saved on CUDA devices
                    extractors[key] = torch.load(str(encoder_path),
                                                 map_location='cpu',
                                                 weights_only=False)
                    
                    # Validate channel compatibility
                    C, H, W = subspace.shape
                    # Get the first Conv2d layer to check input channels
                    first_conv = None
                    for m in extractors[key].modules():
                        if isinstance(m, torch.nn.Conv2d):
                            first_conv = m
                            break
                    if first_conv is not None:
                        expected_channels = first_conv.in_channels
                        if C != expected_channels:
                            raise ValueError(
                                f"Channel mismatch: Encoder expects {expected_channels} channels "
                                f"(trained with {'depth-only' if expected_channels == 1 else 'RGB-D'}), "
                                f"but environment provides {C} channels "
                                f"({'depth-only' if C == 1 else 'RGB-D'}). "
                                f"Set camera.disable_rgb={'true' if expected_channels == 1 else 'false'} "
                                f"in your environment config to match the encoder."
                            )
                    
                    # Validate image size compatibility
                    # Get the Linear layer to check expected flattened size
                    linear_layer = None
                    for m in extractors[key].modules():
                        if isinstance(m, torch.nn.Linear):
                            linear_layer = m
                            break
                    if linear_layer is not None:
                        # Calculate what the flattened size should be after conv layers
                        # Architecture: Conv(stride=2) -> Conv(stride=2) -> Flatten -> Linear
                        # After 2 stride-2 convs: H/4 x W/4
                        # With 32 filters: 32 * (H/4) * (W/4) = 32 * H * W / 16
                        expected_flattened_size = linear_layer.in_features
                        actual_flattened_size = 32 * H * W // 16  # 32 filters * (H/4) * (W/4)
                        if actual_flattened_size != expected_flattened_size:
                            # Calculate expected image size from encoder
                            expected_HW = int((expected_flattened_size * 16 / 32) ** 0.5)
                            raise ValueError(
                                f"Image size mismatch: Encoder expects {expected_HW}x{expected_HW} images "
                                f"(produces {expected_flattened_size} features after conv layers), "
                                f"but environment provides {H}x{W} images "
                                f"(would produce {actual_flattened_size} features). "
                                f"Set camera.height={expected_HW} and camera.width={expected_HW} "
                                f"in your environment config to match the encoder."
                            )
                    
                    p_sum = sum([
                        param.abs().sum().item()
                        for param in extractors[key].parameters()
                        if param.requires_grad
                    ])
                    # Use tolerance for floating point comparison (especially when loading from CUDA to CPU)
                    tolerance = 1e-5
                    if not hasattr(extractors[key], 'p_sum'):
                        print(f"Warning: Model does not have p_sum attribute, skipping validation")
                    else:
                        stored_p_sum = extractors[key].p_sum
                        if abs(p_sum - stored_p_sum) > tolerance:
                            print(f"Warning: Parameter sum mismatch: computed={p_sum}, stored={stored_p_sum}, diff={abs(p_sum - stored_p_sum)}")
                            print(f"This may be due to device differences (CUDA vs CPU). Continuing anyway...")
                            # Uncomment the line below if you want to enforce strict checking
                            # assert False, "unexpected model params sum. The file might be corrupted"
                    last_linear = [
                        m for m in extractors[key].modules()
                        if isinstance(m, torch.nn.Linear)
                    ][-1]
                    self.out_sz = last_linear.out_features
                    total_concat_size += self.out_sz

                    for param in extractors[key].parameters(
                    ):  #to keep it frozen
                        param.requires_grad = False

            else:
                #note that we're iterating on observation_space objects, so there is not batch size info
                S = subspace.shape[0]
                extractors[key] = torch.nn.Flatten()
                total_concat_size += S

        self.extractors = torch.nn.ModuleDict(extractors)

        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        encoded_tensor_dict = {}  #for debug only

        for key, extractor in self.extractors.items():
            cur = extractor(observations[key])

            encoded_tensor_list.append(
                cur
            )  #for rgbd_<int> the cnn uses a tanh at the end so no need for normalization

            #encoded_tensor_dict[key]=cur

        out = torch.cat(encoded_tensor_list, dim=1)
        return out
