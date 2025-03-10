﻿// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using TensorStack.Florence.Common;

namespace TensorStack.Florence.Processing
{
    public interface ITokenProcessor
    {
        /// <summary>
        /// Processes the specified token results.
        /// </summary>
        /// <param name="tokenResult">The token result.</param>
        /// <returns>System.Boolean[].</returns>
        public bool[] Process(BeamResult[] tokenResult);
    }
}
