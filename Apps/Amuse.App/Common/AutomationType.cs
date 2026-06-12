using System.ComponentModel.DataAnnotations;

namespace Amuse.App.Common
{
    public enum AutomationType
    {

        [Display(Name = "Seed Batch", Description = "Generate the specified amount of times with a unique seed value")]
        Seed = 0,

        [Display(Name = "Prompt Lines", Description = "Generate for each prompt line in the specified text file")]
        PromptLines = 10,

        [Display(Name = "Prompt Files", Description = "Generate for each prompt in a text file in the specified folder")]
        PromptFiles = 11,

        [Display(Name = "Input Files", Description = "Generate for each input file in the specified folder")]
        InputFiles = 12,
    }
}