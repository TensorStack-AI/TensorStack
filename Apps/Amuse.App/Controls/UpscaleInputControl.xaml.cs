using Amuse.App.Common;
using Amuse.App.Views;
using System.Threading.Tasks;
using System.Windows;
using TensorStack.WPF;
using TensorStack.WPF.Controls;

namespace Amuse.App.Controls
{
    /// <summary>
    /// Interaction logic for UpscaleInputControl.xaml
    /// </summary>
    public partial class UpscaleInputControl : BaseControl
    {
        private UpscaleInputOption _selectedOption;

        /// <summary>
        /// Initializes a new instance of the <see cref="UpscaleInputControl"/> class.
        /// </summary>
        public UpscaleInputControl()
        {
            InitializeComponent();
        }

        public static readonly DependencyProperty PipelineProperty = DependencyProperty.Register(nameof(Pipeline), typeof(PipelineModel), typeof(UpscaleInputControl), new PropertyMetadata<UpscaleInputControl, PipelineModel>((c, o, n) => c.OnPipelineChanged(o, n)));
        public static readonly DependencyProperty OptionsProperty = DependencyProperty.Register(nameof(Options), typeof(UpscaleInputOptions), typeof(UpscaleInputControl));
        public static readonly DependencyProperty AutomationOptionsProperty = DependencyProperty.Register(nameof(AutomationOptions), typeof(AutomationOptions), typeof(UpscaleInputControl));
        public static readonly DependencyProperty IsExecutingProperty = DependencyProperty.Register(nameof(IsExecuting), typeof(bool), typeof(UpscaleInputControl));
        public static readonly DependencyProperty IsAutomatingProperty = DependencyProperty.Register(nameof(IsAutomating), typeof(bool), typeof(UpscaleInputControl));
        public static readonly DependencyProperty AutomationProgressProperty = DependencyProperty.Register(nameof(AutomationProgress), typeof(ProgressInfo), typeof(UpscaleInputControl));

        public View ViewType { get; set; }
        public PipelineModel Pipeline
        {
            get { return (PipelineModel)GetValue(PipelineProperty); }
            set { SetValue(PipelineProperty, value); }
        }


        public UpscaleInputOptions Options
        {
            get { return (UpscaleInputOptions)GetValue(OptionsProperty); }
            set { SetValue(OptionsProperty, value); }
        }

        public AutomationOptions AutomationOptions
        {
            get { return (AutomationOptions)GetValue(AutomationOptionsProperty); }
            set { SetValue(AutomationOptionsProperty, value); }
        }

        public ProgressInfo AutomationProgress
        {
            get { return (ProgressInfo)GetValue(AutomationProgressProperty); }
            set { SetValue(AutomationProgressProperty, value); }
        }

        public bool IsExecuting
        {
            get { return (bool)GetValue(IsExecutingProperty); }
            set { SetValue(IsExecutingProperty, value); }
        }

        public bool IsAutomating
        {
            get { return (bool)GetValue(IsAutomatingProperty); }
            set { SetValue(IsAutomatingProperty, value); }
        }

        public UpscaleInputOption SelectedOption
        {
            get { return _selectedOption; }
            set { SetProperty(ref _selectedOption, value); }
        }


        private Task OnPipelineChanged(PipelineModel oldPipeline, PipelineModel newPipeline)
        {
            if (newPipeline is null || newPipeline.UpscaleModel is null)
                return Task.CompletedTask;

            var previousOptions = Options;
            var oldModel = oldPipeline?.UpscaleModel;
            var newModel = newPipeline.UpscaleModel;
            var newOptions = newModel.DefaultOptions;

            if (oldModel == newModel)
                return Task.CompletedTask;

            // UpscaleModel
            if (newModel is not null)
            {
                Options = new UpscaleInputOptions
                {
                    IsTileEnabled = newOptions.IsTileEnabled,
                    TileSize = newOptions.TileSize,
                    TileOverlap = newOptions.TileOverlap,
                };

                AutomationOptions = new AutomationOptions
                {
                    ViewType = ViewType,
                    UseInputSize = true,
                    Type = AutomationType.InputFiles,
                };
            }
            return Task.CompletedTask;
        }

    }


    public enum UpscaleInputOption
    {
        Options = 0,
        Advanced = 1,
        Automation = 2,
    }
}