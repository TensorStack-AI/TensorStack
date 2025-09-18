// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Linq;
using TensorStack.Common.Tensor;
using TensorStack.Common.Vision;

namespace TensorStack.TextGeneration.Pipelines.Florence
{
    public class PreProcessor
    {
        private readonly FlorenceConfig _configuration;
        private readonly CoordinateScaler _coordinateScaler;
        private readonly CLIPImageOptions _clipFeatureOptions;

        /// <summary>
        /// Initializes a new instance of the <see cref="PreProcessor"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public PreProcessor(FlorenceConfig configuration)
        {
            _configuration = configuration;
            _coordinateScaler = new CoordinateScaler(_configuration.ImageContextWidth, _configuration.ImageContextHeight);
            _clipFeatureOptions = new CLIPImageOptions(configuration.ImageSampleSize, configuration.ImageSampleSize);
        }


        /// <summary>
        /// Processes the GenerateOptions to extract TaskPrompt and CLIP PixelValues.
        /// </summary>
        /// <param name="options">The options.</param>
        public string ProcessPrompt(FlorenceOptions options)
        {
            var regionTokens = ConvertCoordinates(options.Region, options.Image);
            var prompt = options.TaskType switch
            {
                TaskType.OCR => "What is the text in the image?",
                TaskType.OCR_WITH_REGION => "What is the text in the image, with regions?",
                TaskType.CAPTION => "What does the image describe?",
                TaskType.DETAILED_CAPTION => "Describe in detail what is shown in the image.",
                TaskType.MORE_DETAILED_CAPTION => "Describe with a paragraph what is shown in the image.",
                TaskType.OD => "Locate the objects with category name in the image.",
                TaskType.DENSE_REGION_CAPTION => "Locate the objects in the image, with their descriptions.",
                TaskType.REGION_PROPOSAL => "Locate the region proposals in the image.",
                TaskType.CAPTION_TO_PHRASE_GROUNDING => $"Locate the phrases in the caption: {options.Prompt}",
                TaskType.REFERRING_EXPRESSION_SEGMENTATION => $"Locate {options.Prompt} in the image with mask",
                TaskType.REGION_TO_SEGMENTATION => $"What is the polygon mask of region {regionTokens}",
                TaskType.OPEN_VOCABULARY_DETECTION => $"Locate {options.Prompt} in the image.",
                TaskType.REGION_TO_CATEGORY => $"What is the region {regionTokens}?",
                TaskType.REGION_TO_DESCRIPTION => $"What does the region {regionTokens} describe?",
                TaskType.REGION_TO_OCR => $"What text is in the region {regionTokens}?",
                _ => options.Prompt
            };
            return prompt;
        }


        public ImageTensor ProcessImage(FlorenceOptions options)
        {
            return CLIPImage.Process(options.Image, _clipFeatureOptions);
        }


        /// <summary>
        /// Converts the specified region to Prompt token values `loc_xxx`
        /// </summary>
        /// <param name="region">The region.</param>
        /// <param name="sourceImage">The source image.</param>
        /// <returns>System.String.</returns>
        public string ConvertCoordinates(CoordinateBox<float> region, ImageTensor sourceImage)
        {
            if (region == default)
                return null;

            var scaledRegion = _coordinateScaler.ScaleDown([region], sourceImage).First();
            return $"<loc_{scaledRegion.MinX}><loc_{scaledRegion.MinY}><loc_{scaledRegion.MaxX}><loc_{scaledRegion.MaxY}>";
        }

    }
}