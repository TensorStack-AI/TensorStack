// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Collections.Generic;
using System.Linq;
using TensorStack.Florence.Common;

namespace TensorStack.Florence.Processing
{
    public class EOSTokenProcessor : ITokenProcessor
    {
        private readonly HashSet<long> _eosTokenId;

        /// <summary>
        /// Initializes a new instance of the <see cref="EOSTokenProcessor"/> class.
        /// </summary>
        /// <param name="eosTokenId">The eos token identifier.</param>
        public EOSTokenProcessor(long eosTokenId)
        {
            _eosTokenId = [eosTokenId];
        }


        /// <summary>
        /// Processes the specified token result.
        /// </summary>
        /// <param name="tokenResult">The token result.</param>
        /// <returns>System.Boolean[].</returns>
        public bool[] Process(BeamResult[] tokenResult)
        {
            var output = new bool[tokenResult.Length];
            for (int i = 0; i < tokenResult.Length; i++)
            {
                output[i] = tokenResult[i].Tokens.Count > 2 && tokenResult[i].Tokens[2..].Any(_eosTokenId.Contains);
            }
            return output;
        }
    }
}