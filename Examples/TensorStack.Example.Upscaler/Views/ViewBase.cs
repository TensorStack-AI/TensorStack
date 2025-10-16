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
        }

        public Settings Settings { get; }
    }

    public enum View
    {
        ImageUpscale = 1,
        VideoUpscale = 2,
        Interpolation = 3,
    }
}
