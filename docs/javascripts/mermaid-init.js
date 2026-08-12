document$.subscribe(() => {
  if (typeof mermaid === "undefined") return;

  mermaid.initialize({
    startOnLoad: false,
    theme: "dark",
    securityLevel: "loose",
  });

  const blocks = document.querySelectorAll("pre code.language-mermaid");
  for (const block of blocks) {
    const parent = block.parentElement;
    if (!parent || parent.dataset.mermaidRendered === "true") continue;
    const graphDefinition = block.textContent || "";
    const container = document.createElement("div");
    container.className = "mermaid";
    container.textContent = graphDefinition;
    parent.replaceWith(container);
    parent.dataset.mermaidRendered = "true";
  }

  mermaid.run({ querySelector: ".mermaid" });
});
