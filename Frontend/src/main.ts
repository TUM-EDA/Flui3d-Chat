import { createApp } from 'vue'
import App from './App.vue'
import './assets/main.css'
import Toast, { POSITION } from 'vue-toastification'
import 'vue-toastification/dist/index.css'
import { library } from '@fortawesome/fontawesome-svg-core'
import { FontAwesomeIcon } from '@fortawesome/vue-fontawesome'
import {
  faFlask,
  faCircleExclamation,
  faArrowRight,
  faComments,
  faRotateLeft,
  faImages,
  faAngleRight,
  faAngleLeft,
  faGear,
  faDownload,
  faDatabase,
} from '@fortawesome/free-solid-svg-icons'

import { faLightbulb } from '@fortawesome/free-regular-svg-icons'

library.add(
  faFlask,
  faCircleExclamation,
  faArrowRight,
  faComments,
  faRotateLeft,
  faImages,
  faAngleRight,
  faAngleLeft,
  faGear,
  faLightbulb,
  faDownload,
  faDatabase,
)

const app = createApp(App)
app.component('font-awesome-icon', FontAwesomeIcon)
app.use(Toast, {
  position: POSITION.TOP_RIGHT,
})
app.mount('#app')
