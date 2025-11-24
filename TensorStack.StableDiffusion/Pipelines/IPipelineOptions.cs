// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using TensorStack.Common.Pipeline;
using TensorStack.Common.Tensor;
using TensorStack.StableDiffusion.Enums;
using TensorStack.StableDiffusion.Models;

namespace TensorStack.StableDiffusion.Pipelines
{
    public interface IPipelineOptions : IRunOptions
    {
        int Seed { get; set; }
        int Width { get; set; }
        int Height { get; set; }

        string Prompt { get; set; }
        string NegativePrompt { get; set; }
        float GuidanceScale { get; set; }
        float GuidanceScale2 { get; set; }
        public SchedulerType Scheduler { get; set; }

        float Strength { get; set; }
        ImageTensor InputImage { get; set; }


        ControlNetModel ControlNet { get; set; }
        float ControlNetStrength { get; set; }
        ImageTensor InputControlImage { get; set; }

        int ClipSkip { get; set; }
        float AestheticScore { get; set; }
        float AestheticNegativeScore { get; set; }

        bool IsLowMemoryEnabled { get; set; }
        bool IsLowMemoryComputeEnabled { get; set; }
        bool IsLowMemoryEncoderEnabled { get; set; }
        bool IsLowMemoryDecoderEnabled { get; set; }
        bool IsLowMemoryTextEncoderEnabled { get; set; }
        bool IsPipelineCacheEnabled { get; set; }

        bool HasControlNet => ControlNet is not null;
        bool HasInputImage => InputImage is not null;
        bool HasInputControlImage => InputControlImage is not null;
    }
}
