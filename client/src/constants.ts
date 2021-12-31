export const MAX_IMAGE_SIZE = 1024

// import image0Url from './assets/startImage0.jpeg'
import image1Url from './assets/startImage1.jpeg'
import image2Url from './assets/startImage2.jpeg'
import image3Url from './assets/startImage3.jpeg'
import image4Url from './assets/startImage4.jpeg'
import image5Url from './assets/startImage5.jpeg'
import image6Url from './assets/startImage6.jpeg'

export const DEFAULT_IMAGES = [
    image1Url,
    image2Url,
    image3Url,
    image6Url,
    image4Url,
    image5Url
]

const serverPort = import.meta.env.VITE_SERVER_PORT || 8004
const host = window.location.hostname;
const hasPort = window.location.host.endsWith(":8003") || window.location.host.endsWith(":8004");
const port = hasPort ? `:${serverPort}` : "";
const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
export const SERVER_URL = `${proto}//${host}${port}/ws`;
