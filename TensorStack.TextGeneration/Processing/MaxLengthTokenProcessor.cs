// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
namespace TensorStack.TextGeneration.Processing
{

    public class MaxLengthTokenProcessor : ITokenProcessor
    {
        private readonly int _maxLength;

        /// <summary>
        /// Initializes a new instance of the <see cref="MaxLengthTokenProcessor"/> class.
        /// </summary>
        /// <param name="maxLength">The maximum length.</param>
        public MaxLengthTokenProcessor(int maxLength)
        {
            _maxLength = maxLength;
        }


        /// <summary>
        /// Processes the specified beam search result.
        /// </summary>
        /// <param name="beamSearchResult">The beam search result.</param>
        /// <returns>System.Boolean[].</returns>
        public bool Process(Sequence beamSearchResult)
        {
            return beamSearchResult.Tokens.Count >= _maxLength;
        }
    }
}
