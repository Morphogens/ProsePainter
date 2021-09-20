import { writable, readable, get } from 'svelte/store'

// export const socket = new WebSocket('ws://192.168.193.42:9999')
// export const socket = new WebSocket('ws://localhost:8005/ws')
export let socket = new WebSocket('ws://morphogen-1:8005/ws')

// preserve the socket across HMR updates
if (import.meta.hot) {
    if (import.meta.hot.data.stores) {
      socket = import.meta.hot.data.socket
    }
    import.meta.hot.accept()
    import.meta.hot.dispose(() => {
      import.meta.hot.data.socket = socket
    })
  }
  
  
export const socketOpen = readable(false, (set) => {    
    set(socket.readyState === WebSocket.OPEN);
    socket.addEventListener('open', (event) => {
        console.log('SOCKET CONNECTED');
        set(true)
    })    
    socket.addEventListener('close', (event) => {
        console.log('SOCKET DISCONNECTED');
        set(false)
    })
})

export function messageServer(message:string, data:any) {
    if (get(socketOpen)) {
        socket.send(JSON.stringify({ message, data }))
    }
}