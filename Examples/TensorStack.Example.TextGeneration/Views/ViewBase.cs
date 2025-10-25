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
        TextSummary = 0
    }
}
