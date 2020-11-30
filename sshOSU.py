import pexpect


def scpfiles(cmd): 
	var_password = 'Denise&Harley1389*' 
	var_command = cmd 
	var_child = pexpect.spawn(var_command) 
	i = var_child.expect(['kessler.363@ast-tycho.asc.ohio-state.edu\'s password:',pexpect.EOF]) 
	if i==0: 
		var_child.sendline(var_password) 
		var_child.expect(pexpect.EOF) 
