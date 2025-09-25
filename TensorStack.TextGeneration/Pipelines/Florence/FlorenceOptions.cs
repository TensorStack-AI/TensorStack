// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.ComponentModel.DataAnnotations;
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


    public record FlorenceSearchOptions : FlorenceOptions
    {
        public FlorenceSearchOptions() { }
        public FlorenceSearchOptions(FlorenceOptions options) : base(options) { }
    }


    public enum TaskType
    {
        [Display(Name = "None", Description = "Free text prompt without a predefined task.")]
        NONE,

        [Display(Name = "OCR", Description = "Reads all the visible text in an image.")]
        OCR,

        [Display(Name = "OCR with Region", Description = "Reads text in an image and also gives the region where each piece of text is found.")]
        OCR_WITH_REGION,

        [Display(Name = "Caption", Description = "Generates a short description of the overall image.")]
        CAPTION,

        [Display(Name = "Detailed Caption", Description = "Produces a richer description of the image with more detail than a normal caption.")]
        DETAILED_CAPTION,

        [Display(Name = "More Detailed Caption", Description = "Gives a very thorough and verbose description of the image.")]
        MORE_DETAILED_CAPTION,

        [Display(Name = "Object Detection", Description = "Identifies and localizes objects in the image with bounding boxes.")]
        OD,

        [Display(Name = "Dense Region Caption", Description = "Splits the image into many regions and generates a caption for each region.")]
        DENSE_REGION_CAPTION,

        [Display(Name = "Caption to Phrase Grounding", Description = "Finds the specific region(s) in the image that correspond to a given phrase in a caption.")]
        CAPTION_TO_PHRASE_GROUNDING,

        [Display(Name = "Referring Expression Segmentation", Description = "Given a phrase (e.g., 'the red car'), segments out that exact object region at the pixel level.")]
        REFERRING_EXPRESSION_SEGMENTATION,

        [Display(Name = "Region to Segmentation", Description = "Converts a region (bounding box) into a precise pixel-level segmentation mask.")]
        REGION_TO_SEGMENTATION,

        [Display(Name = "Open Vocabulary Detection", Description = "Detects objects of arbitrary categories, even ones not seen during training.")]
        OPEN_VOCABULARY_DETECTION,

        [Display(Name = "Region to Category", Description = "Assigns a category label (e.g., 'cat', 'chair') to a given region.")]
        REGION_TO_CATEGORY,

        [Display(Name = "Region to Description", Description = "Generates a natural-language description for a given region.")]
        REGION_TO_DESCRIPTION,

        [Display(Name = "Region to OCR", Description = "Extracts text only from within a specified region.")]
        REGION_TO_OCR,

        [Display(Name = "Region Proposal", Description = "Suggests candidate regions of interest in the image (without labeling them).")]
        REGION_PROPOSAL
    }
}
