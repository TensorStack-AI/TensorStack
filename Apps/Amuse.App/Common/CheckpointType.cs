using System.ComponentModel.DataAnnotations;

namespace Amuse.App.Common
{
    public enum CheckpointType
    {

        [Display(Name = "Local Folder")]
        LocalFolder = 0,

        [Display(Name = "Local File")]
        LocalFile = 1,

        [Display(Name = "Online Folder")]
        OnlineFolder = 20,

        [Display(Name = "Online File")]
        OnlineFile = 21,

        [Display(Name = "Component")]
        Component = 100,
    }
}
