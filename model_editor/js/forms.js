
function get_default_form_values(form) {
  const obj = {}
  for (let item of form.elements) {
    if (item.tagName == "INPUT" || item.tagName == "SELECT") {
      if (item.type == "checkbox")
        obj[item.name] = item.defaultChecked
      else
        obj[item.name] = item.defaultValue
    }
  }
  return obj
}

function get_form_values(form, obj) {
  for (let item of form.elements) {
    if (item.tagName == "INPUT" || item.tagName == "SELECT") {
      if (item.type == "checkbox")
        obj[item.name] = item.checked
      else
        obj[item.name] = item.value
    }
  }
  return obj
}

function set_form_values(form, obj) {
  for (let item of form.elements) {
    if (item.tagName == "INPUT" || item.tagName == "SELECT") {
      const value = obj ? obj[item.name] : undefined
      if (value == undefined) {
        if (item.type == "checkbox")
          item.checked = item.defaultChecked
        else
          item.value = item.defaultValue
      }
      else {
        if (item.type == "checkbox")
          item.checked = value
        else
          item.value = value
      }
    }
  }
}

function enable_form(form) {
  for (let item of form.elements) {
    item.disabled = false
  }
}

function disable_form(form) {
  for (let item of form.elements) {
    item.disabled = true
  }
}
