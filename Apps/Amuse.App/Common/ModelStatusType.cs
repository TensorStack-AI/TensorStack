using System.ComponentModel.DataAnnotations;

namespace Amuse.App.Common
{
    public enum ModelStatusType
    {
        [Display(Name = "Available")]
        Available = 0,

        [Display(Name = "Unknown")]
        Unknown = 1,

        [Display(Name = "Downloading")]
        Downloading = 11,

        [Display(Name = "Download Queued")]
        DownloadQueue = 12,

        [Display(Name = "Download Failed")]
        DownloadFailed = 13,

        [Display(Name = "Installed")]
        Installed = 20,
    }
}
