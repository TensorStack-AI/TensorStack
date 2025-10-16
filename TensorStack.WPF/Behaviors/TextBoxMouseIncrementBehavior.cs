using Microsoft.Xaml.Behaviors;
using System.Windows.Controls;
using System.Windows.Input;

namespace TensorStack.WPF.Behaviors
{
    public class TextBoxMouseIncrementBehavior : Behavior<TextBox>
    {
        /// <summary>
        /// Handles the PreviewMouseWheel event of the AssociatedObject control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="MouseWheelEventArgs"/> instance containing the event data.</param>
        private void AssociatedObject_PreviewMouseWheel(object sender, MouseWheelEventArgs e)
        {
            var textbox = (TextBox)sender;
            if (e.Delta > 0)
            {
                if (int.TryParse(textbox.Text, out var result))
                {
                    textbox.Text = $"{++result}";
                }
            }
            else
            {
                if (int.TryParse(textbox.Text, out var result))
                {
                    textbox.Text = $"{--result}";
                }
            }
        }


        /// <summary>
        /// Called after the behavior is attached to an AssociatedObject.
        /// </summary>
        /// <remarks>
        /// Override this to hook up functionality to the AssociatedObject.
        /// </remarks>
        protected override void OnAttached()
        {
            base.OnAttached();
            AssociatedObject.PreviewMouseWheel += AssociatedObject_PreviewMouseWheel;
        }


        /// <summary>
        /// Called when the behavior is being detached from its AssociatedObject, but before it has actually occurred.
        /// </summary>
        /// <remarks>
        /// Override this to unhook functionality from the AssociatedObject.
        /// </remarks>
        protected override void OnDetaching()
        {
            base.OnDetaching();
            AssociatedObject.PreviewMouseWheel -= AssociatedObject_PreviewMouseWheel;
        }
    }
}
