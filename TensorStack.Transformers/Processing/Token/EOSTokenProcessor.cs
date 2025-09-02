// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Collections.Generic;
using System.Linq;

namespace TensorStack.Transformers.Processing
{
    public class EOSTokenProcessor : ITokenProcessor
    {
        private readonly int _minLength;
        private readonly HashSet<long> _eosTokenId;

        /// <summary>
        /// Initializes a new instance of the <see cref="EOSTokenProcessor"/> class.
        /// </summary>
        /// <param name="eosTokenId">The eos token identifier.</param>
        public EOSTokenProcessor(int minLength, params long[] eosTokenIds)
        {
            _minLength = minLength;
            _eosTokenId = [.. eosTokenIds];
        }


        /// <summary>
        /// Processes the specified token result.
        /// </summary>
        /// <param name="tokenResult">The token result.</param>
        /// <returns>System.Boolean.</returns>
        public bool Process(Sequence tokenResult)
        {
            var eosTokenFound = tokenResult.Tokens.Count > 2 && tokenResult.Tokens[2..].Any(_eosTokenId.Contains);
            if (eosTokenFound)
            {
                if (tokenResult.Length < _minLength)
                {
                    tokenResult.Score = float.NegativeInfinity;
                }
            }
            return eosTokenFound;
        }
    }
}