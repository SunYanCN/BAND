/*
 * Initialize animation and Material Design effects for Buttons and Text Fields
 *
 * Author: Sai Hemanth Bheemreddy
 */

var MDCButtons = document.querySelectorAll('.mdc-button');
for(var i = 0; i < MDCButtons.length; i++)
  mdc.ripple.MDCRipple.attachTo(MDCButtons[i]);

var MDCTextFields = document.querySelectorAll('.mdc-text-field');
for(var i = 0; i < MDCTextFields.length; i++)
  mdc.textField.MDCTextField.attachTo(MDCTextFields[i]);