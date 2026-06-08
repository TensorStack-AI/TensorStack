using System.Collections.Generic;
using TensorStack.Common;

namespace TensorStack.Python.Config
{
    public record EnvironmentConfig
    {
        public bool IsDebug { get; set; }
        public string Directory { get; set; }
        public string Environment { get; set; }
        public string PythonVersion { get; set; }
        public string[] Requirements { get; set; }
        public Dictionary<string, string> Variables { get; set; }


        public readonly static string[] DefaultRequirements =
        [
            "typing==3.7.4.3",
            "wheel==0.46.3",
            "transformers==5.5.4",
            "accelerate==1.13.0",
            "diffusers@https://github.com/huggingface/diffusers/archive/f3d42be118f9af7ed9697b686fba09a8bdcd71d1.zip",
            "protobuf==7.34.1",
            "sentencepiece==0.2.1",
            "ftfy==6.3.1",
            "scipy==1.17.1",
            "peft==0.19.1",
            "hf-xet==1.4.3",
            "torchsde==0.2.6",
            "gguf==0.18.0",
            "av==17.0.1",
            "optimum-quanto==0.2.7",
            "bitsandbytes==0.49.2",
            "soundfile==0.13.1"
        ];


        public static EnvironmentConfig VendorDefault(VendorType vendorType)
        {
            return vendorType switch
            {
                VendorType.AMD => DefaultROCM,
                VendorType.Nvidia => DefaultCUDA,
                _ => DefaultCPU
            };
        }


        public readonly static EnvironmentConfig DefaultCPU = new()
        {
            PythonVersion = "3.12",
            Environment = "default-cpu",
            Directory = "PythonRuntime",
            Requirements = [
                "torch==2.9.1",
                "torchvaudio==2.9.1",
                "torchvision==0.24.1",
                ..DefaultRequirements,
            ]
        };


        public readonly static EnvironmentConfig DefaultCUDA = new()
        {
            PythonVersion = "3.12",
            Environment = "default-cuda",
            Directory = "PythonRuntime",
            Variables = new Dictionary<string, string> {
                {"CUDA_VISIBLE_DEVICES", "0,1" },
                {"DIFFUSERS_GGUF_CUDA_KERNELS", "true" }
            },
            Requirements =
            [
                "--extra-index-url https://download.pytorch.org/whl/cu128",
                "torch==2.9.1+cu128",
                "torchaudio==2.9.1+cu128",
                "torchvision==0.24.1+cu128",
                ..DefaultRequirements,
            ]
        };


        public readonly static EnvironmentConfig DefaultROCM = new()
        {
            PythonVersion = "3.12",
            Environment = "default-rocm",
            Directory = "PythonRuntime",
            Variables = new Dictionary<string, string> {
                {"MIOPEN_FIND_MODE", "2" },
                {"HIP_VISIBLE_DEVICES", "0,1" },
                {"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1" }
            },
            Requirements =
            [
                "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz",
                "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl",
                "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl",
                "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl",
                "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
                "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
                "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
                ..DefaultRequirements,
            ]
        };

    }
}
