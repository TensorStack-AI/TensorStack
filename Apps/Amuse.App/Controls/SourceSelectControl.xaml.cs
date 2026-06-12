using System.Windows;
using TensorStack.WPF.Controls;

namespace Amuse.App.Controls
{
    /// <summary>
    /// Interaction logic for SourceSelectControl.xaml
    /// </summary>
    public partial class SourceSelectControl : BaseControl
    {
        private VerticalAlignment _textVerticalAlignment = VerticalAlignment.Center;
        private HorizontalAlignment _textHorizontalAlignment = HorizontalAlignment.Center;

        /// <summary>
        /// Initializes a new instance of the <see cref="SourceSelectControl"/> class.
        /// </summary>
        public SourceSelectControl()
        {
            InitializeComponent();
        }

        public static readonly DependencyProperty IsMultiSelectProperty = DependencyProperty.Register(nameof(IsMultiSelect), typeof(bool), typeof(SourceSelectControl));
        public static readonly DependencyProperty IsSourceRequiredProperty = DependencyProperty.Register(nameof(IsSourceRequired), typeof(bool), typeof(SourceSelectControl), new PropertyMetadata(true));
        public static readonly DependencyProperty SourceCountProperty = DependencyProperty.Register(nameof(SourceCount), typeof(int), typeof(SourceSelectControl), new PropertyMetadata(4));
        public static readonly DependencyProperty IsSource1SelectedProperty = DependencyProperty.Register(nameof(IsSource1Selected), typeof(bool), typeof(SourceSelectControl));
        public static readonly DependencyProperty IsSource2SelectedProperty = DependencyProperty.Register(nameof(IsSource2Selected), typeof(bool), typeof(SourceSelectControl));
        public static readonly DependencyProperty IsSource3SelectedProperty = DependencyProperty.Register(nameof(IsSource3Selected), typeof(bool), typeof(SourceSelectControl));
        public static readonly DependencyProperty IsSource4SelectedProperty = DependencyProperty.Register(nameof(IsSource4Selected), typeof(bool), typeof(SourceSelectControl));
        public static readonly DependencyProperty SourceText1Property = DependencyProperty.Register(nameof(SourceText1), typeof(string), typeof(SourceSelectControl), new PropertyMetadata("Source 1"));
        public static readonly DependencyProperty SourceText2Property = DependencyProperty.Register(nameof(SourceText2), typeof(string), typeof(SourceSelectControl), new PropertyMetadata("Source 2"));
        public static readonly DependencyProperty SourceText3Property = DependencyProperty.Register(nameof(SourceText3), typeof(string), typeof(SourceSelectControl), new PropertyMetadata("Source 3"));
        public static readonly DependencyProperty SourceText4Property = DependencyProperty.Register(nameof(SourceText4), typeof(string), typeof(SourceSelectControl), new PropertyMetadata("Source 4"));

        public bool IsMultiSelect
        {
            get { return (bool)GetValue(IsMultiSelectProperty); }
            set { SetValue(IsMultiSelectProperty, value); }
        }

        public bool IsSourceRequired
        {
            get { return (bool)GetValue(IsSourceRequiredProperty); }
            set { SetValue(IsSourceRequiredProperty, value); }
        }

        public int SourceCount
        {
            get { return (int)GetValue(SourceCountProperty); }
            set { SetValue(SourceCountProperty, value); }
        }

        public bool IsSource1Selected
        {
            get { return (bool)GetValue(IsSource1SelectedProperty); }
            set { SetValue(IsSource1SelectedProperty, value); }
        }

        public bool IsSource2Selected
        {
            get { return (bool)GetValue(IsSource2SelectedProperty); }
            set { SetValue(IsSource2SelectedProperty, value); }
        }

        public bool IsSource3Selected
        {
            get { return (bool)GetValue(IsSource3SelectedProperty); }
            set { SetValue(IsSource3SelectedProperty, value); }
        }

        public bool IsSource4Selected
        {
            get { return (bool)GetValue(IsSource4SelectedProperty); }
            set { SetValue(IsSource4SelectedProperty, value); }
        }

        public string SourceText1
        {
            get { return (string)GetValue(SourceText1Property); }
            set { SetValue(SourceText1Property, value); }
        }

        public string SourceText2
        {
            get { return (string)GetValue(SourceText2Property); }
            set { SetValue(SourceText2Property, value); }
        }

        public string SourceText3
        {
            get { return (string)GetValue(SourceText3Property); }
            set { SetValue(SourceText3Property, value); }
        }

        public string SourceText4
        {
            get { return (string)GetValue(SourceText4Property); }
            set { SetValue(SourceText4Property, value); }
        }

        public VerticalAlignment TextVerticalAlignment
        {
            get { return _textVerticalAlignment; }
            set { SetProperty(ref _textVerticalAlignment, value); }
        }

        public HorizontalAlignment TextHorizontalAlignment
        {
            get { return _textHorizontalAlignment; }
            set { SetProperty(ref _textHorizontalAlignment, value); }
        }


        private void Source1_CheckedChanged(object sender, RoutedEventArgs e)
        {
            if (IsMultiSelect)
            {
                IsSource2Selected = IsSource2Selected && IsSource1Selected;
                IsSource3Selected = IsSource3Selected && IsSource1Selected;
                IsSource4Selected = IsSource4Selected && IsSource1Selected;
            }
            else
            {
                if (IsSource1Selected)
                {
                    IsSource2Selected = false;
                    IsSource3Selected = false;
                    IsSource4Selected = false;
                }
            }

        }


        private void Source2_CheckedChanged(object sender, RoutedEventArgs e)
        {
            if (IsMultiSelect)
            {
                IsSource3Selected = IsSource3Selected && IsSource2Selected;
                IsSource4Selected = IsSource4Selected && IsSource2Selected;
                if (IsSource2Selected)
                {
                    IsSource1Selected = true;
                }
            }
            else
            {
                if (IsSource2Selected)
                {
                    IsSource1Selected = false;
                    IsSource3Selected = false;
                    IsSource4Selected = false;
                }
            }
        }


        private void Source3_CheckedChanged(object sender, RoutedEventArgs e)
        {
            if (IsMultiSelect)
            {
                IsSource4Selected = IsSource4Selected && IsSource3Selected;
                if (IsSource3Selected)
                {
                    IsSource1Selected = true;
                    IsSource2Selected = true;
                }
            }
            else
            {
                if (IsSource3Selected)
                {
                    IsSource1Selected = false;
                    IsSource2Selected = false;
                    IsSource4Selected = false;
                }
            }
        }


        private void Source4_CheckedChanged(object sender, RoutedEventArgs e)
        {
            if (IsMultiSelect)
            {
                if (IsSource4Selected)
                {
                    IsSource1Selected = true;
                    IsSource2Selected = true;
                    IsSource3Selected = true;
                }
            }
            else
            {
                if (IsSource4Selected)
                {
                    IsSource1Selected = false;
                    IsSource2Selected = false;
                    IsSource3Selected = false;
                }
            }
        }
    }
}
