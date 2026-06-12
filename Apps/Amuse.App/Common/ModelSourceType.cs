using System.ComponentModel.DataAnnotations;

namespace Amuse.App.Common
{
    public enum ModelSourceType
    {
        [Display(Name = "Local File", Description = "Setup a model from a single safetensor or gguf file")]
        LocalFile = 0,

        [Display(Name = "Local Folder", Description = "Setup a model from a diffusers style folder")]
        LocalFolder = 1,

        [Display(Name = "Checkpoint", Description = "Setup the model components manually")]
        Checkpoint = 2
    }
}
