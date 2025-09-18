// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using TensorStack.Common.Tensor;
using TensorStack.Common.Vision;
using TensorStack.TextGeneration.Common;

namespace TensorStack.TextGeneration.Pipelines.Florence
{
    public record FlorenceOptions : GenerateOptions
    {
        public TaskType TaskType { get; set; }
        public ImageTensor Image { get; set; }
        public CoordinateBox<float> Region { get; set; }
    }

    public record FlorenceSearchOptions : FlorenceOptions;




    public enum TaskType
    {
        NONE,
        OCR,
        OCR_WITH_REGION,
        CAPTION,
        DETAILED_CAPTION,
        MORE_DETAILED_CAPTION,
        OD,
        DENSE_REGION_CAPTION,
        CAPTION_TO_PHRASE_GROUNDING,
        REFERRING_EXPRESSION_SEGMENTATION,
        REGION_TO_SEGMENTATION,
        OPEN_VOCABULARY_DETECTION,
        REGION_TO_CATEGORY,
        REGION_TO_DESCRIPTION,
        REGION_TO_OCR,
        REGION_PROPOSAL
    }
}
