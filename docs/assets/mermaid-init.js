// Initialize mermaid for both light and dark Material themes.
document.addEventListener("DOMContentLoaded", function () {
  if (typeof mermaid === "undefined") {
    return;
  }
  mermaid.initialize({
    startOnLoad: true,
    theme: "default",
    securityLevel: "loose",
  });
});
