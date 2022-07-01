class Selector {
  // Manages a list of items, where the user can select one item at a time.
  // Items must have a "name" attribute which will be displayed.
  constructor(selector_box, keep_sorted=true) {
    selector_box.classList.add("selector-box")
    const selector_list = document.createElement("div")
    selector_list.classList.add("selector-list")
    selector_box.appendChild(selector_list)
    this.outer = selector_box
    this.inner = selector_list
    this.keep_sorted = keep_sorted
    this.items = []
    this.index = -1
    this._make_selector()
  }

  select(obj) {
    alert("Abstract method called!")
  }

  deselect(obj) {
    alert("Abstract method called!")
  }

  get_selected() {
    // Deselect and reselect the current item to force the subclass to gather
    // all of its parameters.
    const index = this.index
    this._deselect()
    this._select(index)
    return this.items[this.index]
  }

  select_index(index) {
    if (index == this.index)
      return
    this._deselect()
    this._select(index)
  }

  select_name(name) {
    for (const index in this.items) {
      if (this.items[index].name == name) {
        this.select_index(index)
        return true
      }
    }
    return false
  }

  move_up() {
    if (this.index < 1)
      return
    // Swap the selected item up one row.
    const tmp = segment_list.items[this.index - 1]
    segment_list.items[this.index - 1] = segment_list.items[this.index]
    segment_list.items[this.index] = tmp
    // 
    this.index -= 1
    this._make_selector()
  }

  move_down() {
    if (this.index < 0)
      return
    if (this.index + 1 >= this.items.length)
      return
    // Swap the selected item down one row.
    const tmp = segment_list.items[this.index + 1]
    segment_list.items[this.index + 1] = segment_list.items[this.index]
    segment_list.items[this.index] = tmp
    // 
    this.index += 1
    this._make_selector()
  }

  _deselect() {
    if (this.index >= 0) {
      this.deselect(this.items[this.index])
      this.inner.children[this.index].classList.remove("selected")
      this.index = -1
    }
  }

  _select(index) {
    this.index = index
    if (this.index >= 0) {
      this.select(this.items[this.index])
      this.inner.children[this.index].classList.add("selected")
    }
  }

  _make_selector() {
    if (this.keep_sorted) {
      if (typeof this.keep_sorted == 'function')
        this.items.sort((a, b) => this.keep_sorted)
      else
        this.items.sort((a, b) => (a.name < b.name ? -1 : 1))
    }
    const selector = this.inner
    // Remove the existing buttons.
    while (selector.lastChild) {
      selector.removeChild(selector.lastChild)
    }
    // Create new buttons.
    const this_ = this
    for (const [index, item] of this.items.entries()) {
      const btn = document.createElement('button')
      btn.innerText = item.name
      btn.className = "selector-list-btn"
      btn.type = "button"
      btn.onclick = function(event) {this_.select_index(index)}
      selector.appendChild(btn)
    }
    // Apply selected styling.
    if (this.index >= 0) {
      this.inner.children[this.index].classList.add("selected")
    }
    // Hide the selector if there is nothing in it.
    if (this.items.length == 0)
      selector.style.display = "none"
    else
      selector.style.display = "" // Reset display style to initial value.
  }

  add_item(name, item) {
    // Argument "name" can be either a string or a text input field.
    if(typeof name === "object") {
      const input = name
      name = input.value.trim()
      input.value = ""
      if (name == "") {
        input.focus()
        return false
      }
    }
    else {
      name = name.trim()
      if (name == "")
        return false
    }
    // Just select the name if it's already defined.
    if (this.select_name(name))
      return true
    this._deselect()
    // Insert the new item and select it.
    item.name = name
    this.items.push(item)
    this._make_selector()
    this.select_name(name)
    return true
  }

  remove_item(require_confirmation=true) {
    if (this.index < 0)
      return
    const index = this.index
    const name = this.items[index].name
    if (require_confirmation && !confirm(`Confirm delete "${name}"?`))
      return
    this._deselect()
    const [removed] = this.items.splice(index, 1)
    this._make_selector()
    return removed
  }

  set_items(items) {
    this._deselect()
    this.items = items ? Array.from(items) : []
    this._make_selector()
  }

  get_items() {
    this._deselect()
    return Array.from(this.items)
  }

  hide() {
    this.outer.style.display = "none"
  }

  show() {
    this.outer.style.display = ""
  }
}
