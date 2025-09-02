// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Collections.Generic;
using TensorStack.Common.Tensor;
using TensorStack.StableDiffusion.Enums;
using TensorStack.StableDiffusion.Models;
using TensorStack.StableDiffusion.Pipelines;
using TensorStack.StableDiffusion.Schedulers;

namespace TensorStack.StableDiffusion.Common
{
    public record GenerateOptions : IPipelineOptions, ISchedulerOptions
    {
        #region IPipelineOptions

        public int Seed { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }

        public string Prompt { get; set; }
        public string NegativePrompt { get; set; }
        public float GuidanceScale { get; set; }
        public float GuidanceScale2 { get; set; }
        public SchedulerType Scheduler { get; set; }

        public float Strength { get; set; } = 1;
        public float ControlNetStrength { get; set; } = 0.8f;


        public ImageTensor InputImage { get; set; }
        public ImageTensor InputControlImage { get; set; }
        public ControlNetModel ControlNet { get; set; }


        public VideoTensor InputVideo { get; set; }


        public int ClipSkip { get; set; }
        public float AestheticScore { get; set; } = 6f;
        public float AestheticNegativeScore { get; set; } = 2.5f;


        public bool IsLowMemoryEnabled { get; set; }
        public bool IsLowMemoryComputeEnabled { get; set; }
        public bool IsLowMemoryEncoderEnabled { get; set; }
        public bool IsLowMemoryDecoderEnabled { get; set; }
        public bool IsLowMemoryTextEncoderEnabled { get; set; }


        public bool HasControlNet => ControlNet is not null;
        public bool HasInputImage => InputImage is not null;
        public bool HasInputControlImage => InputControlImage is not null;

        #endregion

        #region ISchedulerOptions

        public int Steps { get; set; } = 30;
        public int Steps2 { get; set; } = 10;
        public int TrainTimesteps { get; set; } = 1000;
        public float BetaStart { get; set; } = 0.00085f;
        public float BetaEnd { get; set; } = 0.012f;
        public float[] TrainedBetas { get; set; }
        public TimestepSpacingType TimestepSpacing { get; set; } = TimestepSpacingType.Linspace;
        public BetaScheduleType BetaSchedule { get; set; } = BetaScheduleType.ScaledLinear;
        public int StepsOffset { get; set; } = 0;
        public bool UseKarrasSigmas { get; set; } = false;
        public VarianceType VarianceType { get; set; } = VarianceType.FixedSmall;
        public float SampleMaxValue { get; set; } = 1.0f;
        public bool Thresholding { get; set; } = false;
        public bool ClipSample { get; set; } = false;
        public float ClipSampleRange { get; set; } = 1f;
        public PredictionType PredictionType { get; set; } = PredictionType.Epsilon;
        public AlphaTransformType AlphaTransformType { get; set; } = AlphaTransformType.Cosine;
        public float MaximumBeta { get; set; } = 0.999f;
        public List<int> Timesteps { get; set; }
        public int TrainSteps { get; set; } = 50;
        public float Shift { get; set; } = 1f;

        #endregion
    }
}
