namespace TensorStack.Python.Config
{
    public record EnvironmentConfig
    {
        public bool IsDebug { get; set; }
        public string Environment { get; set; }
        public string[] Requirements { get; set; }
        public string Directory { get; set; }


        public readonly static EnvironmentConfig DefaultCPU = new()
        {
            Environment = "default-cpu",
            Directory = "PythonRuntime",
            Requirements =
            [
                "torchvision==0.22.0",

                 // Default Packages
                "typing",
                "wheel",
                "transformers",
                "accelerate",
                "diffusers",
                "protobuf",
                "sentencepiece",
                "pillow",
                "ftfy",
                "scipy",
                "peft",
                "pillow"
            ]
        };


        public readonly static EnvironmentConfig DefaultCUDA = new()
        {
            Environment = "default-cuda",
            Directory = "PythonRuntime",
            Requirements =
            [
                "--extra-index-url https://download.pytorch.org/whl/cu118",
                "torchvision==0.22.0+cu118",

                 // Default Packages
                "typing",
                "wheel",
                "transformers",
                "accelerate",
                "diffusers",
                "protobuf",
                "sentencepiece",
                "pillow",
                "ftfy",
                "scipy",
                "peft",
                "pillow"
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
                "pillow",
                "ftfy",
                "scipy",
                "peft",
                "pillow"
            ]
        };
    }
}
