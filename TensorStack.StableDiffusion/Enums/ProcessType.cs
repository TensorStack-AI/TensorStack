// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.ComponentModel.DataAnnotations;

namespace TensorStack.StableDiffusion.Enums
{
    public enum ProcessType
    {
        [Display(Name = "Text-To-Image")]
        TextToImage = 0,

        [Display(Name = "Image-To-Image")]
        ImageToImage = 1
    }
}
