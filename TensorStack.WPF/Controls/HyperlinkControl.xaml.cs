using TensorStack.WPF.Services;
using System;
using System.Threading.Tasks;
using System.Windows;

namespace TensorStack.WPF.Controls
{
    /// <summary>
    /// Interaction logic for HyperlinkControl.xaml
    /// </summary>
    public partial class HyperlinkControl : BaseControl
    {
        public HyperlinkControl()
        {
            NavigateLinkCommand = new AsyncRelayCommand(NavigateLink);
            InitializeComponent();
        }

        public static readonly DependencyProperty LinkProperty =
            DependencyProperty.Register("Link", typeof(string), typeof(HyperlinkControl));

        public static readonly DependencyProperty LabelProperty =
            DependencyProperty.Register("Label", typeof(string), typeof(HyperlinkControl));

        public static readonly DependencyProperty IsUnderlineEnabledProperty =
            DependencyProperty.Register("IsUnderlineEnabled", typeof(bool), typeof(HyperlinkControl), new PropertyMetadata(true));

        public string Link
        {
            get { return (string)GetValue(LinkProperty); }
            set { SetValue(LinkProperty, value); }
        }

        public string Label
        {
            get { return (string)GetValue(LabelProperty); }
            set { SetValue(LabelProperty, value); }
        }

        public bool IsUnderlineEnabled
        {
            get { return (bool)GetValue(IsUnderlineEnabledProperty); }
            set { SetValue(IsUnderlineEnabledProperty, value); }
        }

        public AsyncRelayCommand NavigateLinkCommand { get; }

        private async Task NavigateLink()
        {
            try
            {
                URL.NavigateToUrl(Link);
            }
            catch (Exception ex)
            {
                await DialogService.ShowErrorAsync("Navigate Error", $"Failed to navigate to URL: {ex.Message}");
            }
        }
    }
}
