function disable_step_validation() {
  for (const item of document.getElementsByTagName("input")) {
    if (item.type != "number") continue
    if (item.getAttribute("data-strict-step") !== null) continue
    item.addEventListener("input", custom_validation)
  }
}

function custom_validation(event) {
  const input = event.target
  const flags = input.validity
  const invalid = (
      flags.badInput ||
      flags.patternMismatch ||
      flags.rangeOverflow ||
      flags.rangeUnderflow ||
      flags.tooLong ||
      flags.tooShort ||
      flags.typeMismatch ||
      flags.valueMissing)

  if (invalid) {
    input.classList.add("custom_invalid")
  }
  else {
    input.classList.remove("custom_invalid")
  }

  if (flags.stepMismatch) {
    input.setCustomValidity(" ") // Non-empty string, showing nothing.
  }
  else {
    input.setCustomValidity("")
  }
}
