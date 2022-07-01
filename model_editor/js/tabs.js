const tabs = []
const tab_buttons = []

function generate_tabs() {
  tabs.push(simulation_tab)
  tabs.push(mechanisms_tab)
  tabs.push(species_tab)
  tabs.push(regions_tab)
  tabs.push(neurons_tab)
  tabs.push(synapses_tab)

  for (const index in tabs) {
    const tabcontent = tabs[index]
    tabcontent.classList.add("tabcontent")
    const btn = document.createElement("button")
    const bold = document.createElement("b")
    bold.innerText = tabcontent.getAttribute("data-tab-name")
    btn.appendChild(bold)
    btn.className = "tablinks"
    btn.onclick = (() => switchTabs(index))
    tabs_bar.appendChild(btn)
    tab_buttons.push(btn)
  }
}

function switchTabs(index) {
  // Hide all tab contents and deactivate all tab buttons.
  for (const tabcontent of tabs) {
    tabcontent.classList.remove("active")
  }
  for (const tablink of tab_buttons) {
    tablink.classList.remove("active")
  }
  tabs[index].classList.add("active") // Show the selected tab.
  tab_buttons[index].classList.add("active") // Highlight the selected tab button.
}
