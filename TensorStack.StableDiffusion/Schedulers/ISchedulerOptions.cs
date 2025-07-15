// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Collections.Generic;
using TensorStack.StableDiffusion.Enums;

namespace TensorStack.StableDiffusion.Schedulers
{
    public interface ISchedulerOptions
    {
        int Seed { get; set; }
        int Steps { get; set; } 
        int Steps2 { get; set; } 
        int TrainTimesteps { get; set; } 
        float BetaStart { get; set; } 
        float BetaEnd { get; set; } 
        float[] TrainedBetas { get; set; }
        TimestepSpacingType TimestepSpacing { get; set; } 
        BetaScheduleType BetaSchedule { get; set; } 
        int StepsOffset { get; set; }
        bool UseKarrasSigmas { get; set; } 
        VarianceType VarianceType { get; set; }
        float SampleMaxValue { get; set; } 
        bool Thresholding { get; set; }
        bool ClipSample { get; set; } 
        float ClipSampleRange { get; set; } 
        PredictionType PredictionType { get; set; }
        AlphaTransformType AlphaTransformType { get; set; }
        float MaximumBeta { get; set; } 
        List<int> Timesteps { get; set; }
        int TrainSteps { get; set; } 
        float Shift { get; set; } 
    }
}
