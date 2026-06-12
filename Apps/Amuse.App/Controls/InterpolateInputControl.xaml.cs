using Amuse.App.Common;
using Amuse.App.Views;
using System.Threading.Tasks;
using System.Windows;
using TensorStack.WPF;
using TensorStack.WPF.Controls;

namespace Amuse.App.Controls
{
    /// <summary>
    /// Interaction logic for InterpolateInputControl.xaml
    /// </summary>
    public partial class InterpolateInputControl : BaseControl
    {
        private InterpolationInputOption _selectedOption;

        /// <summary>
        /// Initializes a new instance of the <see cref="InterpolateInputControl"/> class.
        /// </summary>
        public InterpolateInputControl()
        {
            InitializeComponent();
        }

        public static readonly DependencyProperty PipelineProperty = DependencyProperty.Register(nameof(Pipeline), typeof(PipelineModel), typeof(InterpolateInputControl), new PropertyMetadata<InterpolateInputControl, PipelineModel>((c, o, n) => c.OnPipelineChanged(o, n)));
        public static readonly DependencyProperty OptionsProperty = DependencyProperty.Register(nameof(Options), typeof(InterpolateInputOptions), typeof(InterpolateInputControl));
        public static readonly DependencyProperty AutomationOptionsProperty = DependencyProperty.Register(nameof(AutomationOptions), typeof(AutomationOptions), typeof(InterpolateInputControl));
        public static readonly DependencyProperty IsExecutingProperty = DependencyProperty.Register(nameof(IsExecuting), typeof(bool), typeof(InterpolateInputControl));
        public static readonly DependencyProperty IsAutomatingProperty = DependencyProperty.Register(nameof(IsAutomating), typeof(bool), typeof(InterpolateInputControl));
        public static readonly DependencyProperty AutomationProgressProperty = DependencyProperty.Register(nameof(AutomationProgress), typeof(ProgressInfo), typeof(InterpolateInputControl));

        public View ViewType { get; set; }
        public PipelineModel Pipeline
        {
            get { return (PipelineModel)GetValue(PipelineProperty); }
            set { SetValue(PipelineProperty, value); }
        }

        public InterpolateInputOptions Options
        {
            get { return (InterpolateInputOptions)GetValue(OptionsProperty); }
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

        public InterpolationInputOption SelectedOption
        {
            get { return _selectedOption; }
            set { SetProperty(ref _selectedOption, value); }
        }


        private Task OnPipelineChanged(PipelineModel oldPipeline, PipelineModel newPipeline)
        {
            Options = new InterpolateInputOptions
            {
                Multiplier = 2
            };

            AutomationOptions = new AutomationOptions
            {
                ViewType = ViewType,
                UseInputSize = true,
                Type = AutomationType.InputFiles,
            };
            return Task.CompletedTask;
        }

    }


    public enum InterpolationInputOption
    {
        Options = 0,
        Advanced = 1,
        Automation = 2,
    }
}