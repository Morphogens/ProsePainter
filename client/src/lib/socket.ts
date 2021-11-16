import { writable, readable, get } from 'svelte/store'

const host = window.location.hostname;
const port = window.location.host.endsWith(":8003") ? ":8004" : "";
const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
const serverURL = `${proto}//${host}${port}/ws`;

let socket = new WebSocket(serverURL)
const events = []
export const socketOpen = writable(false)

function attemptConnect() {
    if (get(socketOpen)) {
        return
    }
    console.log('Attempting socket connection...');    
    socket = new WebSocket(serverURL)
    socket.onerror = function(error) {
        console.log('Socket error:', error)
    };
    socket.onopen = () => { 
        console.log('Websocket connected!');
        socketOpen.set(true)
        for (const [name, callback] of events) {
            console.log('registering event', name);
            socket.addEventListener(name, callback) 
        }
    }
    socket.onclose = (closeEvent) => {
        const reason = closeEvent.reason
        console.log('Socket closed:', reason)
        socketOpen.set(false)
        setTimeout(attemptConnect, 1000)
    }
} 


export function messageServer(topic:string, data:any) {
    if (get(socketOpen)) {
        socket.send(JSON.stringify({ topic, data }))
    }
}
export function addEventListener(name, callback) {
    if (get(socketOpen)) {
        socket.addEventListener(name, callback) 
    }
    events.push([name, callback])
}

attemptConnect()


  // // preserve the socket across HMR updates
// if (import.meta.hot) {
//     if (import.meta.hot.data.stores) {
//         socket = import.meta.hot.data.socket
//     }
//     import.meta.hot.accept()
//     import.meta.hot.dispose(() => {
//         import.meta.hot.data.socket = socket
//     })
// }
