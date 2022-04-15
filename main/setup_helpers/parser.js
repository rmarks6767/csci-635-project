const table = ``

const rows = table.split('\n')

const data = []
rows.forEach(row => {
  data.push([])
  row.split(',').forEach(char => {
    data[data.length - 1].push(Number(char))
  })
})

console.log(data)
