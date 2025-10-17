using System;
using TensorStack.WPF.Controls;
using TensorStack.WPF.Services;

namespace TensorStack.Example.Views
{
    public abstract class ViewBase : ViewControl
    {
        public ViewBase(Settings settings, NavigationService navigationService)
            : base(navigationService)
        {
            Settings = settings;
            Progress = new ProgressInfo();
            DownloadCallback = new Progress<double>(OnDownloadProgress);
        }

        public Settings Settings { get; }
        public ProgressInfo Progress { get;  }
        public Progress<double> DownloadCallback { get; set; }

        protected virtual void OnDownloadProgress(double value)
        {
            Progress.Update((int)value, 100);
        }
    }

    public enum View
    {
        ImageExtractor = 0,
        VideoExtractor = 1,
        ImageBackground = 2,
        VideoBackground = 3
    }
}
