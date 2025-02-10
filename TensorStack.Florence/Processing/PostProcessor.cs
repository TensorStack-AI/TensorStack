// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorStack.Common.Tensor;
using TensorStack.Common.Vision;
using TensorStack.Florence.Common;
using TensorStack.Florence.Tokenizer;

namespace TensorStack.Florence.Processing
{
    public class PostProcessor
    {
        private readonly FlorenceTokenizer _tokenizer;
        private readonly FlorenceConfig _configuration;
        private readonly CoordinateScaler _coordinateScaler;

        /// <summary>
        /// Initializes a new instance of the <see cref="PostProcessor"/> class.
        /// </summary>
        /// <param name="tokenizer">The tokenizer.</param>
        /// <param name="configuration">The configuration.</param>
        public PostProcessor(FlorenceConfig configuration, FlorenceTokenizer tokenizer)
        {
            _tokenizer = tokenizer;
            _configuration = configuration;
            _coordinateScaler = new CoordinateScaler(_configuration.ImageContextWidth, _configuration.ImageContextHeight);
        }


        /// <summary>
        /// Post process the supplied tokens.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="tokens">The tokens.</param>
        /// <returns></returns>
        public GenerateResult Process(GenerateOptions options, IReadOnlyCollection<long> tokens)
        {
            var coordinateType = GetCoordinateType(options.TaskType);
            if (coordinateType.HasValue)
                return ExtractCoordinates(tokens, options.Image, coordinateType.Value);

            return new GenerateResult
            {
                TextResult = _tokenizer.Decode(tokens, false)
            };
        }


        /// <summary>
        /// Parses the coordinates from the supplied tokens.
        /// </summary>
        /// <param name="tokens">The tokens.</param>
        /// <param name="sourceImage">The source Imagetensor.</param>
        /// <param name="coordinateType">Type of the coordinate.</param>
        /// <returns></returns>
        private GenerateResult ExtractCoordinates(IReadOnlyCollection<long> tokens, ImageTensor sourceImage, CoordinateType coordinateType)
        {
            var coordinateResults = coordinateType switch
            {
                CoordinateType.Rectangle => ExtractCoordinateBoxes(tokens, sourceImage, coordinateType),
                CoordinateType.Quadrangle => ExtractCoordinateBoxes(tokens, sourceImage, coordinateType),
                CoordinateType.Polygon => ExtractCoordinatePolygons(tokens, sourceImage),
                _ => throw new NotImplementedException()
            };

            return new GenerateResult
            {
                CoordinateResults = coordinateResults,
            };
        }


        /// <summary>
        /// Parses the box from the supplied tokens.
        /// </summary>
        /// <param name="tokens">The tokens.</param>
        /// <param name="sourceImage">The source Imagetensor.</param>
        /// <param name="coordinateType">Type of the coordinate.</param>
        /// <returns></returns>
        private List<CoordinateResult> ExtractCoordinateBoxes(IReadOnlyCollection<long> tokens, ImageTensor sourceImage, CoordinateType coordinateType)
        {
            var chunkSize = (int)coordinateType;
            var currentLabel = new List<long>();
            var currentCoordinates = new List<int>();
            var results = new List<CoordinateResult>();
            foreach (var tokenId in tokens)
            {
                var tok = _tokenizer.IdToToken(tokenId);
                if (!_tokenizer.TryGetCoordinate(tokenId, out var coordinate))
                {
                    if (currentCoordinates.Count > 0)
                    {
                        var label = _tokenizer.Decode(currentLabel, false);
                        foreach (var box in currentCoordinates.Chunk(chunkSize))
                        {
                            if (box.Length != chunkSize)
                                continue;

                            var coordinates = box.Chunk(2).Select(x => new Coordinate<int>(x)).ToArray();
                            var coordinatesScaled = _coordinateScaler.ScaleUp(coordinates, sourceImage);
                            var coordinateBox = _coordinateScaler.ScaleUp([new CoordinateBox<int>(box)], sourceImage);
                            results.Add(new CoordinateResult
                            {
                                Label = label,
                                CoordinateBox = coordinateBox.First(),
                                Coordinates = coordinatesScaled,
                                CoordinateType = coordinateType
                            });
                        }

                        currentLabel.Clear();
                        currentCoordinates.Clear();
                    }

                    currentLabel.Add(tokenId);
                    continue;
                }

                currentCoordinates.Add(coordinate);
            }
            return results;
        }


        /// <summary>
        /// Parses the polygon from the supplied tokens.
        /// </summary>
        /// <param name="tokens">The tokens.</param>
        /// <param name="sourceImage">The source Imagetensor.</param>
        /// <returns></returns>
        public List<CoordinateResult> ExtractCoordinatePolygons(IReadOnlyCollection<long> tokens, ImageTensor sourceImage)
        {
            var currentLabel = new List<long>();
            var currentCoordinates = new List<int>();
            var results = new List<CoordinateResult>();
            foreach (var tokenId in tokens)
            {
                if (!_tokenizer.TryGetCoordinate(tokenId, out var coordinate))
                {
                    if (currentCoordinates.Count > 0)
                    {
                        var coordinates = new List<Coordinate<int>>();
                        var label = _tokenizer.Decode(currentLabel, false);
                        foreach (var position in currentCoordinates.Chunk(2))
                        {
                            if (position.Length != 2)
                                continue;

                            coordinates.Add(new Coordinate<int>(position[0], position[1]));
                        }

                        var scaledCoordinates = _coordinateScaler.ScaleUp(coordinates.ToArray(), sourceImage);
                        var coordinateBox = new CoordinateBox<int>
                        (
                            coordinates.Min(x => x.PosX),
                            coordinates.Min(x => x.PosY),
                            coordinates.Max(x => x.PosX),
                            coordinates.Max(x => x.PosY)
                        );
                        var scaledCoordinateBox = _coordinateScaler.ScaleUp([coordinateBox], sourceImage).First();
                        results.Add(new CoordinateResult
                        {
                            Label = label,
                            CoordinateBox = scaledCoordinateBox,
                            Coordinates = scaledCoordinates,
                            CoordinateType = CoordinateType.Polygon
                        });

                        currentLabel.Clear();
                        currentCoordinates.Clear();
                    }

                    currentLabel.Add(tokenId);
                    continue;
                }

                currentCoordinates.Add(coordinate);
            }

            return results;
        }


        /// <summary>
        /// Gets the type of coordinate.
        /// </summary>
        /// <param name="taskType">Type of the task.</param>
        /// <returns></returns>
        private static CoordinateType? GetCoordinateType(TaskType taskType)
        {
            switch (taskType)
            {
                // Text
                case TaskType.OCR:
                case TaskType.CAPTION:
                case TaskType.DETAILED_CAPTION:
                case TaskType.MORE_DETAILED_CAPTION:
                    return null;

                // Quadrangle
                case TaskType.REGION_TO_OCR:
                case TaskType.OCR_WITH_REGION:
                    return CoordinateType.Quadrangle;

                // Rectangle
                case TaskType.OD:
                case TaskType.DENSE_REGION_CAPTION:
                case TaskType.CAPTION_TO_PHRASE_GROUNDING:
                case TaskType.OPEN_VOCABULARY_DETECTION:
                case TaskType.REGION_PROPOSAL:
                case TaskType.REGION_TO_CATEGORY:
                case TaskType.REGION_TO_DESCRIPTION:
                    return CoordinateType.Rectangle;

                // Polygon
                case TaskType.REFERRING_EXPRESSION_SEGMENTATION:
                case TaskType.REGION_TO_SEGMENTATION:
                    return CoordinateType.Polygon;
                default:
                    break;
            }
            return null;
        }

    }
}