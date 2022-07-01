function get_parameters() {
  return JSON.stringify({
    simulation: get_form_values(simulation_settings, {}),
    mechanisms: mechanism_list.get_items(),
    species:    species_list  .get_items(),
    regions:    region_list   .get_items(),
    neurons:    neuron_list   .get_items(),
    synapses:   synapse_list  .get_items(),
  })
}

function set_parameters(json) {
  const parameters = json ? JSON.parse(json) : {}
  set_form_values(simulation_settings, parameters.simulation)
  mechanism_list.set_items(parameters.mechanisms)
  species_list  .set_items(parameters.species)
  region_list   .set_items(parameters.regions)
  neuron_list   .set_items(parameters.neurons)
  synapse_list  .set_items(parameters.synapses)
}

function auto_save() {
  localStorage.setItem("state", get_parameters())
}

function auto_load() {
  set_parameters(localStorage.getItem("state"))
}

function reset() {
  if (confirm('Create new model?\nThis will discard the current model.')) {
    set_parameters()
  }
}

function save() {
  const blob = new Blob([get_parameters()], {type : 'application/json'})
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = "model.json"
  link.click()
  URL.revokeObjectURL(url)
}

function load(file) {
  const reader = new FileReader()
  reader.onload = (event => set_parameters(event.target.result))
  reader.readAsText(file)
}
