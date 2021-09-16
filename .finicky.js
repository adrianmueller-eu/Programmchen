// Generate at https://finicky-kickstart.vercel.app/
// Documentation: https://github.com/johnste/finicky/wiki/Configuration

module.exports = {
  defaultBrowser: "Firefox",
  handlers: [
    {
      match: /^https?:\/\/.*aalto.*/,
      browser: "Safari"
    },
  ],
  options: {
    // Hide the finicky icon from the top bar. Default: false
    hideIcon: true,
  }
}
