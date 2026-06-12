using Amuse.App.Common;
using System.Collections.ObjectModel;
using System.Threading.Tasks;
using System.Windows;
using TensorStack.Common;
using TensorStack.WPF;
using TensorStack.WPF.Controls;

namespace Amuse.App.Controls
{
    /// <summary>
    /// Interaction logic for DiffusionCheckpointControl.xaml
    /// </summary>
    public partial class DiffusionCheckpointControl : BaseControl
    {
        private int _selectedIndex;

        /// <summary>
        /// Initializes a new instance of the <see cref="DiffusionCheckpointControl"/> class.
        /// </summary>
        public DiffusionCheckpointControl()
        {
            Components = new ObservableCollection<CheckpointComponent>();
            ComputeCheckpointTypes = [CheckpointType.LocalFolder, CheckpointType.OnlineFolder];
            TextEncodedCheckpointTypes = [CheckpointType.LocalFolder, CheckpointType.OnlineFolder, CheckpointType.Component];
            InitializeComponent();
        }

        public static readonly DependencyProperty SettingsProperty = DependencyProperty.Register(nameof(Settings), typeof(Settings), typeof(DiffusionCheckpointControl));
        public static readonly DependencyProperty CheckpointProperty = DependencyProperty.Register(nameof(Checkpoint), typeof(CheckpointModel), typeof(DiffusionCheckpointControl), new PropertyMetadata<DiffusionCheckpointControl, CheckpointModel>((c, o, n) => c.OnCheckpointChanged(o, n)));
        public static readonly DependencyProperty BackendProperty = DependencyProperty.Register(nameof(Backend), typeof(BackendType), typeof(DiffusionCheckpointControl));
        public ObservableCollection<CheckpointComponent> Components { get; }
        public CheckpointType[] ComputeCheckpointTypes { get; }
        public CheckpointType[] TextEncodedCheckpointTypes { get; }

        public Settings Settings
        {
            get { return (Settings)GetValue(SettingsProperty); }
            set { SetValue(SettingsProperty, value); }
        }

        public CheckpointModel Checkpoint
        {
            get { return (CheckpointModel)GetValue(CheckpointProperty); }
            set { SetValue(CheckpointProperty, value); }
        }

        public BackendType Backend
        {
            get { return (BackendType)GetValue(BackendProperty); }
            set { SetValue(BackendProperty, value); }
        }

        public int SelectedIndex
        {
            get { return _selectedIndex; }
            set { SetProperty(ref _selectedIndex, value); }
        }


        private Task OnCheckpointChanged(CheckpointModel previous, CheckpointModel checkpoint)
        {
            Components.Clear();
            if (checkpoint != null)
            {
                foreach (var component in checkpoint.GetComponents())
                {
                    Components.Add(component);
                }

                if (checkpoint.Compute != null)
                    SelectedIndex = 0;
                if (checkpoint.Unet != null)
                    SelectedIndex = 4;
                if (checkpoint.Transformer != null)
                    SelectedIndex = 5;
            }
            return Task.CompletedTask;
        }

    }
}