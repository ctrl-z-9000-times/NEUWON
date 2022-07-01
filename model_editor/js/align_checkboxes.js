// Wrap all checkboxes in <label> elements so that they float to the left side
// of the page, instead of centering which is the default.
function align_checkboxes() {
  let auto_inc = 0
  for (const item of document.getElementsByTagName("input")) {
    if (item.type != "checkbox") continue
    const chk = item.cloneNode()
    const lbl = document.createElement("label")
    chk.id = `_checkbox_${auto_inc}`
    auto_inc += 1
    lbl.htmlFor = chk.id
    lbl.appendChild(chk)
    item.replaceWith(lbl)
  }
}
