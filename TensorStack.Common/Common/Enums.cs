// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.

using System.ComponentModel;

namespace TensorStack.Common
{
    /// <summary>
    /// Enum Optimization
    /// </summary>
    public enum Optimization
    {
        /// <summary>
        /// No Optimizations
        /// </summary>
        None = 0,

        /// <summary>
        /// Basic Optimizations
        /// </summary>
        Basic = 1,

        /// <summary>
        /// Extended Optimizations
        /// </summary>
        Extended = 2,

        /// <summary>
        /// All Optimizations
        /// </summary>
        All = 99
    }


    /// <summary>
    /// Normalization
    /// </summary>
    public enum Normalization
    {
        None = 0,
        ZeroToOne = 1,
        OneToOne = 2,
        MinMaxZeroToOne = 3,
        MinMaxOneToOne = 4
    }


    /// <summary>
    /// ResizeMode
    /// </summary>
    public enum ResizeMode
    {
        /// <summary>
        /// Strech Image
        /// </summary>
        Stretch = 0,

        /// <summary>
        /// Center Crop Image
        /// </summary>
        Crop = 1,

        /// <summary>
        /// LetterBox Center Image
        /// </summary>
        LetterBox = 2
    }


    /// <summary>
    /// Enum ResizeMethod
    /// </summary>
    public enum ResizeMethod
    {
        Bilinear = 0,
        Bicubic = 1
    }
}
