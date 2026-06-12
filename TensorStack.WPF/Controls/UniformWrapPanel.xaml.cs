using System;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;

namespace TensorStack.WPF.Controls
{
    /// <summary>
    /// Interaction logic for UniformWrapPanel.xaml
    /// </summary>
    public partial class UniformWrapPanel : WrapPanel
    {
        public UniformWrapPanel()
        {
            InitializeComponent();
        }

        public static readonly DependencyProperty MinItemWidthProperty =
            DependencyProperty.Register(nameof(MinItemWidth), typeof(int), typeof(UniformWrapPanel));

        public static readonly DependencyProperty MaxItemWidthProperty =
            DependencyProperty.Register(nameof(MaxItemWidth), typeof(int), typeof(UniformWrapPanel), new PropertyMetadata<UniformWrapPanel>(c => c.UpdateItemLayout(c.ActualWidth)));

        public int MinItemWidth
        {
            get { return (int)GetValue(MinItemWidthProperty); }
            set { SetValue(MinItemWidthProperty, value); }
        }


        public int MaxItemWidth
        {
            get { return (int)GetValue(MaxItemWidthProperty); }
            set { SetValue(MaxItemWidthProperty, value); }
        }


        private Task UpdateItemLayout(double containerWidth)
        {
            if (MinItemWidth <= 0)
                return Task.CompletedTask;
            if (MaxItemWidth <= 0)
                return Task.CompletedTask;

            ItemWidth = CalculateItemWidth(containerWidth, MinItemWidth, MaxItemWidth);
            return Task.CompletedTask;
        }


        protected override Size MeasureOverride(Size constraint)
        {
            UpdateItemLayout(constraint.Width);
            return base.MeasureOverride(constraint);
        }

        public double CalculateItemWidth(double containerWidth, double minWidth, double maxWidth)
        {
            int maxItemCount = (int)Math.Floor(containerWidth / minWidth);
            int minItemCount = (int)Math.Ceiling(containerWidth / maxWidth);
            if (minItemCount == 1 && maxItemCount == 0)
                return containerWidth;

            for (int itemCount = maxItemCount; itemCount >= minItemCount; itemCount--)
            {
                double potentialItemWidth = containerWidth / itemCount;

                if (potentialItemWidth >= minWidth && potentialItemWidth <= maxWidth)
                {
                    return potentialItemWidth;
                }
            }

            if (containerWidth / minItemCount < minWidth)
            {
                return minWidth;
            }
            return maxWidth;
        }
    }
}
