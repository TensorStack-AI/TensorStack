using System.Collections.Generic;
using System.Text.Json.Serialization;
using TensorStack.Common;

namespace TensorStack.Python.Config
{
    public record EnvironmentConfig
    {
        public bool IsDebug { get; set; }
        public string Directory { get; set; }
        public string Environment { get; set; }
        public string[] Requirements { get; set; }
        public Dictionary<string, string> Variables { get; set; }


        public readonly static string[] DefaultRequirements =
        [
            "typing==3.7.4.3",
            "wheel==0.45.1",
            "transformers==4.57.3",
            "accelerate==1.12.0",
            "diffusers==0.36.0",
            "protobuf==6.33.2",
            "sentencepiece==0.2.1",
            "ftfy==6.3.1",
            "scipy==1.16.3",
            "peft==0.18.0",
            "hf-xet==1.2.0",
            "torchsde==0.2.6"
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
            Environment = "default-cuda",
            Directory = "PythonRuntime",
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
            Environment = "default-rocm",
            Directory = "PythonRuntime",
            Variables = new Dictionary<string, string> {
                {"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1" }
            },
            Requirements =
            [
                "https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/rocm_sdk_core-0.1.dev0-py3-none-win_amd64.whl",
                "https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/rocm_sdk_devel-0.1.dev0-py3-none-win_amd64.whl",
                "https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/rocm_sdk_libraries_custom-0.1.dev0-py3-none-win_amd64.whl",
                "https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/rocm-0.1.dev0.tar.gz",
                "https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/torch-2.9.0+rocmsdk20251116-cp312-cp312-win_amd64.whl",
                "https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/torchvision-0.24.0+rocmsdk20251116-cp312-cp312-win_amd64.whl",
                ..DefaultRequirements,
            ]
        };

    }
}
