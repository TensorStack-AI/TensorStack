// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Animation;

namespace TensorStack.WPF.Controls
{
    /// <summary>
    /// Interaction logic for FontAwesome.xaml
    /// </summary>
    public partial class FontAwesome : BaseControl
    {
        private readonly Storyboard _spinAnimation;
        private string _iconUnicode = "\uf004";

        /// <summary>
        /// Initializes a new instance of the <see cref="FontAwesome"/> class.
        /// </summary>
        public FontAwesome()
        {
            InitializeComponent();
            _spinAnimation = FindResource("SpinAnimation") as Storyboard;
        }

        public static readonly DependencyProperty SizeProperty = DependencyProperty.Register(nameof(Size), typeof(int), typeof(FontAwesome), new PropertyMetadata(16));
        public static readonly DependencyProperty IconProperty = DependencyProperty.Register(nameof(Icon), typeof(string), typeof(FontAwesome), new PropertyMetadata<FontAwesome>(c => c.OnIconUpdate()));
        public static readonly DependencyProperty ColorProperty = DependencyProperty.Register(nameof(Color), typeof(Brush), typeof(FontAwesome), new PropertyMetadata(Brushes.Black));
        public static readonly DependencyProperty IconStyleProperty = DependencyProperty.Register(nameof(IconStyle), typeof(FontAwesomeIconStyle), typeof(FontAwesome), new PropertyMetadata(FontAwesomeIconStyle.SharpLight));
        public static readonly DependencyProperty IsSpinnerProperty = DependencyProperty.Register(nameof(IsSpinner), typeof(bool), typeof(FontAwesome), new PropertyMetadata((d, e) => { if (d is FontAwesome control) control.OnIsSpinnerChanged(); }));

        /// <summary>
        /// Gets or sets the icon.
        /// </summary>
        public string Icon
        {
            get { return (string)GetValue(IconProperty); }
            set { SetValue(IconProperty, value); }
        }

        /// <summary>
        /// Gets or sets the size.
        /// </summary>
        public int Size
        {
            get { return (int)GetValue(SizeProperty); }
            set { SetValue(SizeProperty, value); }
        }

        /// <summary>
        /// Gets or sets the color.
        /// </summary>
        public Brush Color
        {
            get { return (Brush)GetValue(ColorProperty); }
            set { SetValue(ColorProperty, value); }
        }

        /// <summary>
        /// Gets or sets the icon style.
        /// </summary>
        public FontAwesomeIconStyle IconStyle
        {
            get { return (FontAwesomeIconStyle)GetValue(IconStyleProperty); }
            set { SetValue(IconStyleProperty, IconStyle); }
        }

        /// <summary>
        /// Gets or sets a value indicating whether this instance is spinning icon.
        /// </summary>
        /// <value>
        ///   <c>true</c> if this instance is spinner; otherwise, <c>false</c>.
        /// </value>
        public bool IsSpinner
        {
            get { return (bool)GetValue(IsSpinnerProperty); }
            set { SetValue(IsSpinnerProperty, value); }
        }


        /// <summary>
        /// Gets or sets the unicode icon.
        /// </summary>
        public string IconUnicode
        {
            get { return _iconUnicode; }
            set { SetProperty(ref _iconUnicode, value); }
        }


        /// <summary>
        /// Called when icon is updated.
        /// </summary>
        /// <returns>Task.</returns>
        private Task OnIconUpdate()
        {
            IconUnicode = Icon.Length == 1 ? Icon : char.ConvertFromUtf32(Convert.ToInt32(Icon, 16));
            return Task.CompletedTask;
        }


        /// <summary>
        /// Called when IsSpinner changed.
        /// </summary>
        private void OnIsSpinnerChanged()
        {
            if (_spinAnimation is null)
                return;

            if (IsSpinner)
                _spinAnimation.Begin();
            else if (!IsSpinner)
                _spinAnimation.Stop();
        }
    }

    public enum FontAwesomeIconStyle
    {
        Regular,
        Light,
        Solid,
        Thin,
        Brands,
        Duotone,
        SharpRegular,
        SharpLight,
        SharpSolid,
        SharpThin,
    }
}
