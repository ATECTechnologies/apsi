-- Neovim: start the Apsi language server (live errors, hover, go to definition, completion, formatting)
-- for every .apsi buffer. Turn it off with  vim.g.apsi_lsp = false  in your init.lua.
-- Use another command with  vim.g.apsi_command = { "py", "-3", "-m", "apsi" }.
if vim.g.loaded_apsi_lsp or not vim.lsp or not vim.lsp.start then
  return
end
vim.g.loaded_apsi_lsp = true

vim.api.nvim_create_autocmd("FileType", {
  pattern = "apsi",
  group = vim.api.nvim_create_augroup("apsi_lsp", { clear = true }),
  callback = function(args)
    if vim.g.apsi_lsp == false then
      return
    end
    local cmd = vim.g.apsi_command or { "apsi" }
    if type(cmd) == "string" then
      cmd = { cmd }
    end
    if vim.fn.executable(cmd[1]) ~= 1 then
      return
    end
    local full = vim.list_extend(vim.deepcopy(cmd), { "lsp" })
    local file = vim.api.nvim_buf_get_name(args.buf)
    local dir = file ~= "" and vim.fs.dirname(file) or vim.fn.getcwd()
    local found = vim.fs.find({ "apsi.toml", ".git" }, { upward = true, path = dir })[1]
    vim.lsp.start({
      name = "apsi",
      cmd = full,
      root_dir = found and vim.fs.dirname(found) or dir,
    }, { bufnr = args.buf })
  end,
})
