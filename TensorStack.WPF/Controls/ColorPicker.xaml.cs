using System.Windows;
using System.Windows.Media;

namespace TensorStack.WPF.Controls
{
    /// <summary>
    /// Interaction logic for ColorPicker.xaml
    /// </summary>
    public partial class ColorPicker : BaseControl
    {
        public ColorPicker()
        {
            InitializeComponent();
        }


        public Color SelectedColor
        {
            get { return (Color)GetValue(SelectedColorProperty); }
            set { SetValue(SelectedColorProperty, value); }
        }


        public static readonly DependencyProperty SelectedColorProperty =
            DependencyProperty.Register("SelectedColor", typeof(Color), typeof(ColorPicker), new PropertyMetadata(Colors.Black));

    




        public bool IsPickerOpen
        {
            get { return (bool)GetValue(IsPickerOpenProperty); }
            set { SetValue(IsPickerOpenProperty, value); }
        }


        public static readonly DependencyProperty IsPickerOpenProperty =
            DependencyProperty.Register("IsPickerOpen", typeof(bool), typeof(ColorPicker), new PropertyMetadata(false));





     


    }
}
