using System.Threading.Tasks;
using TensorStack.Example.Views;
using TensorStack.WPF;
using TensorStack.WPF.Controls;
using TensorStack.WPF.Services;

namespace TensorStack.Example
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : WindowMainBase
    {
        public MainWindow(Settings configuration, NavigationService navigation)
        {
            Navigation = navigation;
            NavigateCommand = new AsyncRelayCommand<View>(NavigateAsync, CanNavigate);
            InitializeComponent();

            NavigateCommand.Execute(View.TextSummary);
        }

        public NavigationService Navigation { get; }
        public AsyncRelayCommand<View> NavigateCommand { get; }


        private async Task NavigateAsync(View view)
        {
            await Navigation.NavigateAsync((int)view);
        }

        private bool CanNavigate(View view)
        {
            return true;
        }


        public override void OnDragBegin(DragDropType type)
        {
            base.OnDragBegin(type);
            Navigation.CurrentView.DragDropType = type;
            Navigation.CurrentView.IsDragDrop = true;
        }


        public override void OnDragEnd()
        {
            base.OnDragEnd();
            Navigation.CurrentView.IsDragDrop = false;
            Navigation.CurrentView.DragDropType = DragDropType.None;
        }
    }
}