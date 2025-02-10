// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorStack.Common.Tensor;

namespace TensorStack.Florence.Processing
{
    public class NoRepeatNGramLogitsProcessor : ILogitsProcessor
    {
        private readonly int _noRepeatNgramSize;

        /// <summary>
        /// Initializes a new instance of the <see cref="NoRepeatNGramLogitsProcessor"/> class.
        /// </summary>
        /// <param name="noRepeatNgramSize">Size of the no repeat ngram.</param>
        public NoRepeatNGramLogitsProcessor(int noRepeatNgramSize)
        {
            _noRepeatNgramSize = noRepeatNgramSize;
        }


        /// <summary>
        /// Processes the specified inputs logita.
        /// </summary>
        /// <param name="inputs">The inputs.</param>
        /// <param name="logits">The logits.</param>
        public void Process(List<long>[] inputs, Tensor<float> logits)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                var inputIds = inputs[i];
                var nGramWindow = inputIds.TakeLast(_noRepeatNgramSize - 1).ToList();
                for (int tokenIndex = 0; tokenIndex < inputIds.Count; tokenIndex++)
                {
                    if (IsRepeatingNGram(nGramWindow, tokenIndex, inputIds, _noRepeatNgramSize))
                    {
                        logits[i, tokenIndex] = float.NegativeInfinity;
                    }
                }
            }
        }


        /// <summary>
        /// Determines whether same sequence of tokens appears too frequently.
        /// </summary>
        /// <param name="nGramWindow">The n gram window.</param>
        /// <param name="currentToken">The current token.</param>
        /// <param name="tokenHistory">The token history.</param>
        /// <param name="nGramSize">Size of the n gram.</param>
        private static bool IsRepeatingNGram(List<long> nGramWindow, int currentToken, List<long> tokenHistory, int nGramSize)
        {
            var newNGram = new List<long>(nGramWindow) { currentToken };
            for (int i = 0; i <= tokenHistory.Count - nGramSize; i++)
            {
                var existingNGram = tokenHistory.Skip(i).Take(nGramSize).ToList();
                if (newNGram.SequenceEqual(existingNGram))
                {
                    return true; // Found a repeating n-gram
                }
            }
            return false;
        }
    }
}

