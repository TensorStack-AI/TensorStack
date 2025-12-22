using System;

namespace TensorStack.Python.Config
{
    public record EnvironmentConfig
    {
        public bool IsDebug { get; set; }
        public string Environment { get; set; }
        public string[] Requirements { get; set; }
        public string Directory { get; set; }


        public static EnvironmentConfig Default(VendorType vendorType)
        {
            return vendorType switch
            {
                VendorType.AMD => DefaultROCM,
                VendorType.Nvidia => DefaultCUDA,
                _ => DefaultCPU
            };
        }

        public static EnvironmentConfig Default(object vendor)
        {
            throw new NotImplementedException();
        }

        public readonly static EnvironmentConfig DefaultCPU = new()
        {
            Environment = "default-cpu",
            Directory = "PythonRuntime",
            Requirements =
            [
                "torch==2.9.1",
                "torchvaudio==2.9.1",
                "torchvision==0.24.1",

                 // Default Packages
                "typing",
                "wheel",
                "transformers",
                "accelerate",
                "diffusers",
                "protobuf",
                "sentencepiece",
                "ftfy",
                "scipy",
                "peft"
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

                 // Default Packages
                "typing",
                "wheel",
                "transformers",
                "accelerate",
                "diffusers",
                "protobuf",
                "sentencepiece",
                "ftfy",
                "scipy",
                "peft"
            ]
        };


        public readonly static EnvironmentConfig DefaultROCM = new()
        {
            Environment = "default-rocm",
            Directory = "PythonRuntime",
            Requirements =
            [
                "https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/rocm_sdk_core-0.1.dev0-py3-none-win_amd64.whl",
                "https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/rocm_sdk_devel-0.1.dev0-py3-none-win_amd64.whl",
                "https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/rocm_sdk_libraries_custom-0.1.dev0-py3-none-win_amd64.whl",
                "https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/rocm-0.1.dev0.tar.gz",
                "https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/torch-2.9.0+rocmsdk20251116-cp312-cp312-win_amd64.whl",
                "https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/torchvision-0.24.0+rocmsdk20251116-cp312-cp312-win_amd64.whl",

                // Default Packages
                "typing",
                "wheel",
                "transformers",
                "accelerate",
                "diffusers",
                "protobuf",
                "sentencepiece",
                "ftfy",
                "scipy",
                "peft"
            ]
        };
    }

    public enum VendorType
    {
        Unknown = 0,
        AMD = 4098,
        Nvidia = 4318,
        Intel = 32902
    }
}
