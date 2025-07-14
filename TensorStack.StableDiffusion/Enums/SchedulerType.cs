// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.ComponentModel.DataAnnotations;

namespace TensorStack.StableDiffusion.Enums
{
    public enum SchedulerType
    {
        [Display(Name = "LMS")]
        LMS = 0,

        [Display(Name = "Euler")]
        Euler = 1,

        [Display(Name = "Euler Ancestral")]
        EulerAncestral = 2,



        [Display(Name = "LCM")]
        LCM = 20
    }
}
